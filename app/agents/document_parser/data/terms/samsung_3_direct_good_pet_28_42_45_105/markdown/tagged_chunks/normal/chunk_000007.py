from langchain_core.documents import Document

chunk = Document(
    page_content=('- 2. 갱신계약 : [갱신형] 특별약관의 보험기간이 끝난 후 제도성 특별약관 「4-1.\n'
 '- [갱신형] 특별약관의 자동갱신 특별약관」 에 따라 갱신된 경우를 말합니다.\n'
 '- 3. 갱신일 : [갱신형] 특별약관이 갱신되기 직전 계약(이하 「갱신전 계약」 이라\n'
 '- 합니다)의 보험기간이 끝난 날의 다음 날을 말합니다.\n'
 '⑦ (재가입형) 특별약관 재가입 관련 용어- 1. 최초계약 : 최초로 체결되는 계약을 말합니다.\n'
 '- 2. 재가입계약 : 이 보험의 사업방법서에서 정한 재가입 절차에 따라 재가입된 계약을\n'
 '- 말합니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000007',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
