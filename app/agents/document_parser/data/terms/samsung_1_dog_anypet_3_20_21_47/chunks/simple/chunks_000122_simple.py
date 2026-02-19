from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 흡인 (주사기 등으로 빨아 들이는 것) 2. 천자 (바늘 또는 관을 꽂아 체액, 조직을 뽑아내거나 약물을 주입하는 것) 등의 조치 '
 '3. 미용성형 목적의 수술 4. 검사 및 진단을 위한 수술 (생검, 복강경 검사)\n'
 '제2조(보험금 등의 지급한도)\n'
 '① 회사는 제1조(보상하는 손해)에서 정한 수술비용 확대보장 보험금은 피보험자가 부담한 수술당일치 료비에서 보통약관에서 지급한 '
 '치료비보험금을 차감 후 보상비율(%)을 곱한 금액이며 보험증권에 정한 1일당 수술비확장 보상한도액을 한도로 아래의 산식에 따라 '
 '보상합니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 22},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000122',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
