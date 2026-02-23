from langchain_core.documents import Document

chunk = Document(
    page_content=('관리 하에 직접적인 치료를 목적으로 기구를 사용하여 생체에 절개, 절단, 절제 등의 조작을 가하는 것을 말합니다. 단 수술에서 아래에 '
 '정한 사항은 제외합니다1. 흡인 (주사기 등으로 빨아 들이는 것)- 2. 천자 (바늘 또는 관을 꽂아 체액, 조직을 뽑아내거나 약물을 '
 '주입하는 것) 등의 조치\n'
 '- 3. 미용성형 목적의 수술\n'
 '- 4. 검사 및 진단을 위한 수술 (생검, 복강경 검사)\n'
 '# 제 5조 (보험금의 청구)# ① 보험수익자는 다음의 서류를 제출하고 보험금을 청구하여야 합니다.- 1. 보험금 청구서(회사 양식)'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
