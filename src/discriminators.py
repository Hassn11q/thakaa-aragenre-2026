# -*- coding: utf-8 -*-
"""Expert-linguist contrastive discriminators for all 74 test genres.

Each entry names the concrete SURFACE/CONTENT signal that separates a genre from its family
siblings — dialect phonology, mushaf orthography, isnad, meta book-blurb language, news datelines,
job domain keywords, textbook level, essay topic. Fed to the Gemma judge appended to each
candidate's definition so the judge decides like a human Arabic linguist. Genre-general: derived
from the released definitions + Arabic linguistic knowledge, no dataset labels.
"""

DISCRIMINATORS = {
    # ---- Creative: dialect (songs) vs fuṣḥā register (poetry) ----
    "classical_poetry": "فصحى تراثية قديمة، وزن وقافية، مفردات جزلة قديمة؛ ليست عامية ولا حديثة (تراث).",
    "msa_poetry": "شعر بالفصحى الحديثة: رمزية واستعارة وتأمل وجداني/اجتماعي معاصر؛ ليست عامية.",
    "egyptian_song_lyrics": "أغنية بالعامية المصرية: ده/دى، مش، عايز/عاوز، ازاي، علشان، انت كده.",
    "gulf_song_lyrics": "أغنية باللهجة الخليجية: وايد، شلونك، چذي، أبغى، مب، زين، هاللي.",
    "iraqi_song_lyrics": "أغنية باللهجة العراقية: القاف تُنطق گ/ك (قلبي→كلبي، تقول→تكول)، شلون، شكو ماكو، هسه، اكو، هواي، خوش.",
    "levantine_song_lyrics": "أغنية باللهجة الشامية: هيك، شو، بدي، هلق، منيح، كتير، عم بـ.",
    "sudanese_song_lyrics": "أغنية باللهجة السودانية: داير، كيف، شنو، زول، ياخ.",
    # ---- Religious: Islamic scripture/tradition/commentary vs Christian ----
    "quran": "آيات قرآنية بالرسم العثماني وعلامات الوقف والتشكيل الكامل؛ كلام إلهي بلا إسناد.",
    "hadith": "يبدأ بإسناد: حدثنا/أخبرنا ... عن فلان عن فلان، قال رسول الله ﷺ.",
    "tafsir": "تفسير آيات قرآنية: قوله تعالى، أي، المعنى، قال المفسرون/ابن عباس.",
    "bible": "نص من الكتاب المقدس المسيحي: يسوع، المسيح، الرب، الإنجيل، التلاميذ.",
    "biblical": "شرح/تفسير مسيحي لنصوص الكتاب المقدس ضمن سياق لاهوتي.",
    "coptic_devotional_literature": "أدب تعبدي مسيحي/قبطي: صلاة، تسبحة، قداس، طقوس، تأمل روحي.",
    # ---- Interactive ----
    "app_reviews": "تقييم تطبيق/برنامج: التطبيق، البرنامج، التحديث، نجوم، رائع/سيء جدًا.",
    "book_reviews": "تقييم كتاب/رواية: الكتاب، الكاتب/الكاتبة، الرواية، القراءة، نجوم.",
    "twitter_posts": "منشور اجتماعي قصير مستقل: تفاعلي، إيموجي، تطويل حروف، رأي عابر عن حدث/شعور.",
    "youtube_comments": "تعليق على فيديو/حلقة: الحلقة، الفيديو، القناة، شكرًا، مخاطبة صاحب المحتوى.",
    # ---- Legal ----
    "constitutions": "نص دستوري: مبادئ وحقوق وبنية الدولة، مواد مرقّمة (مادة رقم)، لكل شخص الحق.",
    "contracts": "اتفاق/شروط خدمة وخصوصية بين أطراف: الطرف الأول/الثاني، البيانات، الاشتراك، غالبًا أسماء شركات لاتينية.",
    "resolutions": "قرار رسمي لهيئة/مجلس: بنود مرقّمة (1- تؤيد، 2- تطلب)، الأمين العام، الدولة الطرف، رموز وثائق.",
    # ---- Informative: encyclopedia (reference) ----
    "culture_encyclopedia": "نص موسوعي مرجعي يشرح التقاليد/التراث/العادات بأسلوب تفسيري محايد.",
    "history_encyclopedia": "نص موسوعي مرجعي يشرح أحداثًا وحضارات وشخصيات تاريخية.",
    "math_encyclopedia": "نص موسوعي يشرح مفاهيم/قوانين/رموز رياضية بأسلوب مرجعي.",
    # ---- Informative: news (journalistic, datelines City أ.ف.ب/د.ب.أ:) ----
    "culture_news": "خبر صحفي عن فعاليات/معارض/أدب/تراث ثقافي.",
    "economy_news": "خبر صحفي اقتصادي: أسواق، أسهم، تجارة، مال، استثمار، بنوك.",
    "international_news": "خبر صحفي دولي: شؤون خارجية، دبلوماسية، نزاعات، علاقات بين دول.",
    "local_news": "خبر صحفي محلي: أحداث/مؤسسات/خدمات مجتمعية داخل منطقة/بلد.",
    "religion_news": "خبر صحفي عن أحداث/مؤسسات دينية وشؤون عامة دينية (تقرير خبري لا نص ديني).",
    "sports_news": "خبر صحفي رياضي: مباريات، بطولات، فرق، لاعبون (غالبًا Dateline: مدينة أ.ف.ب/د.ب.أ:).",
    # ---- Informative: book_description (BLURB about a book, meta: يتناول الكتاب/المؤلف) ----
    "arts_book_description": "وصف كتاب عن الفنون/الثقافة البصرية/الإبداع (لغة تعريفية بالكتاب).",
    "biographies_memoirs_book_description": "وصف كتاب سيرة/مذكرات عن حياة شخص/شخصية تاريخية.",
    "economics_business_book_description": "وصف كتاب عن الاقتصاد/المال/الإدارة/الأعمال.",
    "family_children_book_description": "وصف كتاب عن الأسرة/التربية/الطفل.",
    "history_geography_book_description": "وصف كتاب عن التاريخ/الجغرافيا/المجتمعات.",
    "islamic_book_description": "وصف/تعريف بكتاب إسلامي (ميتا: الكتاب، المؤلف، يتناول) وليس نصًا دينيًا مباشرًا.",
    "journalism_media_book_description": "وصف كتاب عن الصحافة/الإعلام/الاتصال.",
    "literature_fiction_book_description": "وصف كتاب أدبي/روائي: حبكة، شخصيات، سرد.",
    "philosophy_book_description": "وصف كتاب فلسفي/فكري: منطق، أخلاق، تقاليد فكرية.",
    "political_book_description": "وصف كتاب سياسي: حكم، أيديولوجيا، شؤون عامة.",
    "science_nature_book_description": "وصف كتاب علمي/طبيعي: علوم، بيئة، أحياء.",
    "sharia_law_book_description": "وصف كتاب عن الفقه/الشريعة/القانون الإسلامي (تعريف بالكتاب لا نص فقهي).",
    "sports_entertainment_book_description": "وصف كتاب عن الرياضة/الترفيه/المشاهير.",
    # ---- Informative: jobs (Job ad 'مطلوب ...' by DOMAIN) ----
    "admin_and_secretarial_jobs": "إعلان وظيفة إدارية/سكرتارية/مكتبية.",
    "automotive_and_mechanics_jobs": "إعلان وظيفة سيارات/ميكانيكا/صيانة/إصلاح.",
    "cleaning_services_jobs": "إعلان وظيفة نظافة/تنظيف/خدمات منزلية.",
    "customer_service_jobs": "إعلان وظيفة خدمة عملاء/دعم/كول سنتر.",
    "drivers_and_delivery_jobs": "إعلان وظيفة سائق/توصيل/لوجستيات.",
    "education_jobs": "إعلان وظيفة تعليم/تدريس/مؤهلات تربوية.",
    "engineering_jobs": "إعلان وظيفة هندسة/تصميم/أنظمة تقنية.",
    "finance_and_accounting_jobs": "إعلان وظيفة مالية/محاسبة/تدقيق.",
    "health_and_beauty_jobs": "إعلان وظيفة تجميل/عناية/سبا.",
    "healthcare_jobs": "إعلان وظيفة رعاية صحية/طب/تمريض/عيادة.",
    "human_resources_jobs": "إعلان وظيفة موارد بشرية/توظيف/إدارة موظفين.",
    "information_technology_jobs": "إعلان وظيفة تقنية معلومات/برمجة/شبكات.",
    "law_and_legal_services_jobs": "إعلان وظيفة قانونية/محاماة/استشارات قضائية.",
    "manufacturing_and_retail_jobs": "إعلان وظيفة تصنيع/إنتاج/بيع بالتجزئة.",
    "marketing_jobs": "إعلان وظيفة تسويق/إعلان/علامة تجارية/حملات ترويج.",
    "media_and_design_jobs": "إعلان وظيفة إعلام/تصميم جرافيك/عمل إبداعي رقمي.",
    "sales_jobs": "إعلان وظيفة مبيعات: مندوب مبيعات، تحقيق أهداف بيع، إقناع العملاء.",
    "security_jobs": "إعلان وظيفة أمن/حراسة/مراقبة/حماية.",
    "technicians_and_craftsmen_jobs": "إعلان وظيفة فنيين/حرفيين/مهن يدوية/صيانة عملية.",
    "tourism_and_restaurants_jobs": "إعلان وظيفة سياحة/ضيافة/مطاعم/خدمة طعام.",
    # ---- Learning: textbooks by LEVEL vs student_writing by TOPIC ----
    "early_education_textbooks": "كتاب مدرسي لمرحلة مبكرة: لغة مبسّطة جدًا، حروف/أرقام، تمارين أولية.",
    "upper_elementary_textbooks": "كتاب مدرسي ابتدائي عليا: شرح مبسّط وتمارين تأسيسية.",
    "intermediate_textbooks": "كتاب مدرسي متوسط: مفاهيم أكاديمية موجّهة لمرحلة متوسطة.",
    "high_school_textbooks": "كتاب مدرسي ثانوي: مفاهيم علمية معقّدة (فيزياء/أحياء/كيمياء)، مصطلحات كثيفة.",
    "communication_student_writing": "مقال طالب عن التواصل/الإعلام/العلاقات الاجتماعية.",
    "culture_student_writing": "مقال طالب عن الثقافة/التراث/الهوية.",
    "national_development_student_writing": "مقال طالب عن التنمية الوطنية/التقدم/المسؤولية المدنية.",
    "schoolwork_student_writing": "واجب/تعبير مدرسي عام لإظهار الفهم (موضوع عام غير متخصص).",
    "social_media_student_writing": "مقال طالب عن وسائل التواصل/السلوك الرقمي/الإنترنت.",
    "sports_student_writing": "مقال طالب عن الرياضة/النشاط البدني/العمل الجماعي.",
    "technology_student_writing": "مقال طالب عن التكنولوجيا/الأدوات الرقمية/الابتكار.",
    "values_student_writing": "مقال طالب عن الأخلاق/القيم/المسؤولية/المبادئ (العدل، الأمانة).",
}
