from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 흡인(吸引)- 2. 천자(穿刺) 등의 조치\n'
 '- 3. 미용성형 목적의 수술\n'
 '- 4. 검사 및 진단을 위한 수술(생검(生檢), 복강경검사(腹腔鏡檢査) 등)\n'
 '116 KB 금쪽같은 펫보험(강아지)(무배당)(26.01)|  |\n'
 '| --- |\n'
 '| 용 어 풀 이 ∙ 절단 : 특정부위를 잘라 내는 것 ∙ 절제 : 특정부위를 잘라 없애는 것 ∙ 흡인 : 주사기 등으로 빨아들이는 것 '
 '|\n'
 '# ∙ 천자 : 바늘 또는 관을 꽂아 체액․조직을 뽑아내거나 약물을 주입하는 것제7조(특별약관의 소멸)'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
