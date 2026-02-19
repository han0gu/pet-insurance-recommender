from langchain_core.documents import Document

chunk = Document(
    page_content=('형] 특별약관의 자동갱신 특별약관」에 따라 갱신된 경우를 말합니다.\n'
 '3. 갱신일: [갱신형] 특별약관이 갱신되기 직전 계약(이하「갱신 전 계약」이라 합니 다)의 보험기간이 끝난 날의 다음 날을 말합니다.\n'
 '⑦ (재가입형) 특별약관 재가입 관련 용어\n'
 '1. 최초계약 : 최초로 체결되는 계약을 말합니다. 2. 재가입계약 : 이 보험의 사업방법서에서 정한 재가입 절차에 따라 재가입된 계약을 '
 '말합니다.\n'
 '제2관 보험금의 지급\n'
 '제 3조 (보험금의 지급사유)'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 32},
 'term_type': 'basic',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000010',
              'chunk_char_len': 249,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
