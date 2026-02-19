from langchain_core.documents import Document

chunk = Document(
    page_content=('① 회사는 보험기간 중에 보험증권에 기재된 반려동물에게 보통약관에서 보상하는 상해 또는 질병이 발생하여 그 치료를 직접적인 목적으로 '
 "국내에서 수의사에게 수술(이하 '사고'라 합니다)을 받은 경 우 수술 당일 발생한 수술비 및 치료비를 보통약관에서 보상하는 치료비보험금에 "
 '추가하여 보상하 여 드립니다'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 22},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000119',
              'chunk_char_len': 166,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
