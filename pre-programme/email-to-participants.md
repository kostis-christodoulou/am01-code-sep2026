# AM01 pre-programme email

Two versions below. Send the short one as the email body and attach (or link) the guide; the long one is only if you want to spell out more.

---

## TL;DR version (recommended email body)

**Subject:** AM01 — set up Python before we start (30–45 min)

Dear all,

Before our first session, please set up Python on your own laptop and make a start on the pre-programme assignment. Budget 30–45 minutes for the setup.

**The short version**

1. Install **uv**, the tool we use to manage Python.
   - Windows, in PowerShell: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
   - macOS, in Terminal: `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - Close the window, open a new one, and check it worked: `uv --version`
2. **Unzip `am01-code.zip`** into a local folder — for example `C:\Code\am01` on Windows or `~/Code/am01` on macOS. Not OneDrive, Dropbox, iCloud, or Google Drive; cloud sync causes real problems here.
3. Open a terminal **in that folder** and run `uv sync`. The first run takes a few minutes: it installs Python 3.12 and every package the course needs.
4. Check it: `uv run python --version` should report 3.12.x.
5. Open the folder in **Antigravity IDE**, then open `pre-programme/your-name-pre-course.ipynb`. Save it under your own name and work through the three tasks. Choose the kernel from the project's `.venv` when asked.

**If you have used Anaconda before:** keep it, but do not mix it with this. If your terminal prompt shows `(base)`, run `conda deactivate` first, and use `uv` commands only inside the AM01 folder.

The attached guide, *Getting Started with Python Programming*, walks through all of this in detail with troubleshooting. If you get stuck, email me a screenshot of the whole terminal window and tell me which step you were on — please don't spend an hour trying random fixes.

You are not expected to produce perfect work on the assignment. Attempt every task, note where you got stuck, and bring those questions to the first session.

Best wishes,
Kostis

---

## Longer version (if you prefer more hand-holding)

**Subject:** AM01 — pre-programme setup and assignment

Dear all,

Welcome to AM01. There are two things to do before our first session: get Python working on your own laptop, and make a start on the pre-programme assignment. Together they take a couple of hours, and the setup part is the fiddly bit — please do it early rather than the night before.

**What is attached**

- `Getting Started with Python Programming.docx` — the full setup guide. Work through it in order.
- `am01-code.zip` — the course files: the package list, the datasets, and your pre-course notebook.

**What you will end up with**

One folder on your laptop containing the course files, a private Python 3.12 environment inside it, and an IDE that opens that same folder. Nothing is installed system-wide, so nothing you already have gets disturbed.

**The four steps**

1. Install **uv**. This is the one tool that manages everything else — it installs Python for you, so you don't need to install Python separately.
2. Create a folder such as `C:\Code\am01` (Windows) or `~/Code/am01` (macOS) and unzip the attachment into it. Please use a local folder, not one synced by OneDrive, Dropbox, iCloud Drive, or Google Drive — sync locks files while Python is using them and produces errors that are hard to diagnose.
3. Open a terminal in that folder and run `uv sync`. This reads the package list and installs Python 3.12 plus around 190 packages. The first run takes a few minutes and looks like a lot of scrolling text; that is normal.
4. Install Antigravity IDE and open that same folder in it. When it asks about agent autonomy, choose review-driven development so you see proposed changes before they run.

**Then the assignment.** In the `pre-programme` folder you will find `your-name-pre-course.ipynb`. Save a copy under your own name, then work through the three tasks: a short bio written in markdown, a country comparison using the gapminder data, and a look at flight delays using the nycflights13 data. Several cells are left deliberately incomplete — you write the code, and you also write short answers in prose where the notebook asks for them. Those written answers matter as much as the code.

**A note for Anaconda users.** You do not need to uninstall Anaconda and your existing projects can stay exactly as they are. But do not mix the two here. If your terminal prompt begins with `(base)`, run `conda deactivate` before you start, and don't run `conda install` or `pip install` while setting up the AM01 environment.

**When to ask for help.** The guide has a troubleshooting section covering the five things that usually go wrong. If none of it helps, email me a screenshot of the entire terminal or IDE window, the command you ran, and whether you are on Windows or macOS. Please ask rather than improvising — an hour of guesswork usually creates a second problem on top of the first.

You are not expected to have everything perfect. Attempt every task, write down where you got stuck, and bring those questions to the first session.

Best wishes,
Kostis
