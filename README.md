# fanam-intern-eval




## 🚀 Detailed Submission Instructions

To ensure isolated evaluation environments, you will use the **Forking Method**. Do not push directly to this repository or open a Pull Request. Follow these exact steps to complete and submit your work:

**Step 1: Fork the Repository**
1. Ensure you are logged into your GitHub account.
2. Navigate to this base repository.
3. Click the **"Fork"** button in the top right corner of the page.
4. Keep the default settings and click **"Create fork"**. This creates an independent copy of the repository under your personal GitHub account.

**Step 2: Clone Your Fork Locally**
1. On your newly forked repository page, click the green **"Code"** button and copy the HTTPS or SSH URL.
2. Open your terminal and run:
   ```bash
   git clone <paste-your-copied-url-here>

```

3. Navigate into the project folder:
```bash
cd <repository-name>

```



**Step 3: Set Up Your Environment**

1. Create a virtual environment to isolate your dependencies:
```bash
python -m venv venv

```


2. Activate the virtual environment:
* **Mac/Linux:** `source venv/bin/activate`
* **Windows:** `venv\Scripts\activate`


3. Install any required dependencies (and remember to generate a `requirements.txt` before submitting).

**Step 4: Create a `.gitignore` File**
Before making your first commit, ensure you do not accidentally upload system files, virtual environments, or secret keys.

1. Create a file named `.gitignore` in the root directory.
2. Add the following standard exclusions:
```text
venv/
__pycache__/
.env
*.pyc
.DS_Store

```



**Step 5: Develop and Commit**

1. Complete the coding task in your local environment.
2. Stage your changes:
```bash
git add .

```


3. Commit your work with descriptive messages explaining *what* and *why*:
```bash
git commit -m "feat: implement database connection and API routing"

```


*(Note: Make multiple logical commits as you build, rather than one massive commit at the end.)*

**Step 6: Push to Your Fork**

1. Push your committed code back to your GitHub account:
```bash
git push origin main

```



**Step 7: Final Submission**

1. Verify that your code, `requirements.txt`, and updated `README.md` (with setup instructions) are visible on your GitHub fork page.
2. **If your fork is Public:** Send the direct URL of your repository to your evaluator.
3. **If your fork is Private:** Go to *Settings > Collaborators > Add people*, invite `[Insert Your GitHub Username Here]`, and then send the URL to your evaluator.

