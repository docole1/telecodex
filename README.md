# TeleCodex

**Use Codex from Telegram to work on projects in your own Linux environment.**

TeleCodex is a self-hosted Telegram bot that connects Telegram conversations to the local Codex app-server. Send tasks from private chats or groups and follow up in Telegram while Codex works on projects in your Linux environment. It brings familiar Codex CLI workflows to Telegram through commands and buttons, though some CLI commands are not available.

Codex CLI and TeleCodex share conversations for projects under `CODEX_WORKSPACE_ROOT`: start a conversation in Codex CLI, then open it with `/chats` in Telegram and continue there. Use `/chats` to switch between project conversations; separate private chats and attached groups have independent Codex conversations, so tasks in different Telegram chats can run at the same time. Conversation history and Codex settings such as model, reasoning effort, and Plan/Default mode stay with each conversation. TeleCodex-specific preferences such as response detail and language stay in the bot. Permissions are also managed by TeleCodex, but they are chosen separately for each conversation.

TeleCodex uses your ChatGPT sign-in through Codex CLI and does not require an OpenAI API key or API usage charges.

**Status: Beta.** Bugs and stalled conversations can occur. If a conversation gets stuck, `/recover` can restart its Codex session while preserving history and settings; if recovery fails, use `/new` to start a separate conversation.

## Requirements

- A Linux environment with Python 3.12 or newer.
- Codex CLI installed and available to the Linux user that will run TeleCodex.
- A ChatGPT account with Codex access, signed in through Codex CLI as that same Linux user.
- A Telegram bot token and the numeric Telegram user IDs allowed to operate the bot.

## Install and run

Clone the repository and install TeleCodex into a virtual environment:

```bash
git clone https://github.com/docole1/telecodex.git
cd telecodex
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -e .
cp .env.example .env
```

Sign in to ChatGPT through Codex CLI as the same Linux user that will run the bot, then verify the login:

```bash
codex login
codex login status
```

Edit `.env` and set at least:

```dotenv
TELEGRAM_BOT_TOKEN=your-bot-token
TELEGRAM_ALLOWED_USER_IDS=1234567
CODEX_WORKSPACE_ROOT=/srv/projects
```

To find your numeric Telegram user ID, open the third-party bot [@getmyid_bot](https://t.me/getmyid_bot) and press **Start**. Copy the numeric ID it shows, not your `@username`. To allow multiple operators, separate their IDs with commas, for example `TELEGRAM_ALLOWED_USER_IDS=1234567,9876543`.

`CODEX_WORKSPACE_ROOT` is the starting directory for `/new` and the initial shell directory. Make sure the service user can read and write the projects under it. Review [`.env.example`](.env.example) for optional proxy, model, audio, logging, and network settings.

Start the bot in the foreground:

```bash
.venv/bin/telecodex run
```

## Updating

When TeleCodex reports a new release, review its release notes. From the repository directory, install that release and restart the bot:

```bash
git fetch --tags
git checkout vX.Y.Z
.venv/bin/pip install -e .
```

Replace `vX.Y.Z` with the release tag, then restart TeleCodex using the same method you use to run it.

## First steps in Telegram

1. Open a private chat with your bot and send `/new`.
2. Choose a project directory with the buttons.
3. Send Codex a task as a regular message. You can send follow-up messages while it is working.
4. Use `/chats` to return to a saved conversation and `/help` to see available commands.

To use TeleCodex in a group, add the bot to the group, run `/group_add`, and configure trigger words.

Notes for groups:

- Only users listed in `TELEGRAM_ALLOWED_USER_IDS` can control the bot in a group. Other members cannot.
- By default, Telegram's privacy mode hides ordinary group messages from bots, so trigger words will not work. In [@BotFather](https://t.me/BotFather), turn privacy mode off with `/setprivacy` or make the bot a group admin, then remove the bot from the group and add it again. Replies to the bot's messages work either way.

## Commands

Send tasks as ordinary messages. Use `/help` or `/start` in Telegram to show the command list there.

### Conversations and projects

| Command | Description |
| --- | --- |
| `/new` | Choose a project directory and start a new Codex conversation. |
| `/chats` | Browse saved conversations, inspect their details, and select or fork one. |
| `/rename <name>` | Rename the current conversation. |
| `/fork` | Create a copy of the current conversation. |
| `/delete` | Delete the current conversation and its forks after confirmation. |
| `/compact` | Queue context compaction after the active task and Codex background terminals finish. |
| `/recap` | Ask Codex for a summary without compacting the conversation. |
| `/init` | Ask Codex to create a project-root `AGENTS.md` if none exists. |
| `/files` | Browse, preview, and download files from the active project. |

### Settings

| Command | Description |
| --- | --- |
| `/model` | Choose a model and its reasoning effort. |
| `/defaults` | Set model, permissions, reply mode, and memory defaults for new chats, plus the default message mode for new groups. |
| `/fast` | Toggle the Fast tier when the selected model supports it. |
| `/mode` | Choose answers-only, answers with silent progress updates, or detailed activity. |
| `/plan [task]` | Switch Plan/Default mode, or start planning a task in Plan mode. |
| `/goal [objective]` | Set or inspect a persistent Goal; supports `edit`, `pause`, `resume`, and `clear`. |
| `/permissions` | Choose the permission mode for the current conversation. |
| `/skills` | Browse, configure, and select a Codex skill for the next task. |
| `/memories` | Configure Codex memory use and generation. |
| `/mcp [verbose]` | Inspect and manage MCP servers; `verbose` shows more details. |
| `/language` | Choose the interface language. |

### Task control

| Command | Description |
| --- | --- |
| `/stop` | Stop the active Codex task or shell command. |
| `/ps` | Inspect or stop background terminals for the current Codex conversation. |
| `/cancel` | Works while editing a goal or answering a text prompt in the MCP setup wizard; it is not a general cancellation command. Use inline buttons for button-based steps. |
| `/shell` | Open an interactive shell in the Linux environment running TeleCodex. |
| `/shell <command>` | Run one command in that Linux environment. |
| `/exit` | Close the interactive shell and return to Codex. |

### Diagnostics

| Command | Description |
| --- | --- |
| `/status` | Show the selected conversation, model, permissions, context, and usage limits. |
| `/doctor` | Check Telegram, Codex, authentication, audio, and polling health. |
| `/recover` | Restart Codex for the current conversation if it hangs or fails, preserving its history and settings. |

### Telegram groups

| Command | Description |
| --- | --- |
| `/group_add` | Attach the current group and configure trigger words. |
| `/group_remove` | Detach the current group. |
| `/group_mode` | Choose trigger/reply mode or forward every ordinary message to Codex. |

## Access and permissions

TeleCodex runs as the Linux user that starts the process. Commands sent through `/shell` run as that user and are not isolated by Codex's command sandbox.

`/permissions` chooses how Codex handles permissions in the current conversation:

- **Ask**: Codex works in the project and sends approval requests to the Telegram chat.
- **Auto-review**: Codex evaluates permission requests itself.
- **Full access**: removes Codex's restrictions and disables the command sandbox.
- **Read-only**: Codex cannot write files.

Until you choose a mode, Codex's own configured policy applies.

Sandboxed permission modes disable network access for shell commands by default. Codex's built-in web search and MCP connections still work.

## License

MIT — see [LICENSE](LICENSE).
