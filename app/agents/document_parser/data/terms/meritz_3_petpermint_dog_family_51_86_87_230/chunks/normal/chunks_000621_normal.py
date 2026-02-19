from langchain_core.documents import Document

chunk = Document(
    page_content=('제14조(특별약관의 자동갱신)\n'
 '\uf000 배상책임 관련 특별약관이 다음 각 호의 조건을 충족하 고, 계약이 끝나는 날의 전일까지 계약자로부터 별도의 의 사표시가 없을 '
 '때에는 종전의 계약이 끝나는 날의 다음날 (이하「갱신일」이라 합니다)에 동일한 보장내용으로 갱신 되는 것으로 합니다.\n'
 '① 갱신될 계약(이하「갱신계약」이라 합니다)이 끝나는 날이 회사가 정한 기간내일 것 ② 갱신일에 있어서 반려동물의 만나이가 회사가 정한 '
 '나 이의 범위내일 것 ③ 갱신전 계약의 보험료가 정상적으로 납입완료 되었을 것 ④ 갱신전 계약이 소멸되지 않을 것'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 183},
 'term_type': 'special',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000621',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
