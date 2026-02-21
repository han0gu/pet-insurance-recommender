from langchain_core.documents import Document

chunk = Document(
    page_content=("반려견보험 애니펫보통약관- 3 -당신에게 좋은보험 삼성화재# 제1관 목적 및 용어의 정의# 제1조(목적)이 보험계약(이하 '계약'이라 "
 "합니다)은 보험계약자(이하 '계약자'라 합니다)와 보험회사(이하 '회사'라 합"),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000000',
              'chunk_char_len': 118,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
