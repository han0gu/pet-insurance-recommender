from langchain_core.documents import Document

chunk = Document(
    page_content=('비 용, 상담 수수료, 지도 비용 및 이와 동종의 비용 ⑮ 왕진 비용, 가입동물의 이송비, 동물병원에 가지 않 고 약제만 배달되는 배달료 '
 '및 이와 동종의 비용 ⑯ 안락사 비용, 시체처치 및 해부검사, 장례비, 이장비 등 사후에 필요한 비용 ⑰ 마이크로 칩 이식 비용, 각종 '
 '증빙서류의 작성비용 (우송비 포함) ⑱ 과잉진료행위로 인한 비용'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 158},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000515',
              'chunk_char_len': 188,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
