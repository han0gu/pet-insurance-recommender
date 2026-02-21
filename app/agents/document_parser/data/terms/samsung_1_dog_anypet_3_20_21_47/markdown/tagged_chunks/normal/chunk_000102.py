from langchain_core.documents import Document

chunk = Document(
    page_content=('인 치료를 목적으로 기구를 사용하여 생체에 절개, 절단, 절제 등의 조작을 가하는 것을 말합니다. 단 수술에\n'
 '서 아래에 정한 사항은 제외합니다.- 1. 흡인 (주사기 등으로 빨아 들이는 것)\n'
 '- 2. 천자 (바늘 또는 관을 꽂아 체액, 조직을 뽑아내거나 약물을 주입하는 것) 등의 조치\n'
 '- 3. 미용성형 목적의 수술\n'
 '- 4. 검사 및 진단을 위한 수술 (생검, 복강경 검사)\n'
 '# 제2조(보험금 등의 지급한도)- ① 회사는 제1조(보상하는 손해)에서 정한 슬관절 수술비용보험금은 보험증권에 기재된 보상비율(%)'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'limit', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_000102',
              'chunk_char_len': 284,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
