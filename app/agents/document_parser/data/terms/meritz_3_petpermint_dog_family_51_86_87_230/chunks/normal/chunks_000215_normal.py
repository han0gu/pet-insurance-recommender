from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 제1항의 통지에 따라 위험의 증가로 보험료를 더 내야 할 경우 회사가 청구한 추가보험료(정산금액을 포함합니다) 를 계약자가 '
 '납입하지 않았을 때, 회사는 위험이 증가되기 전에 적용된 보험요율(이하「변경전 요율」이라 합니다)의 위험이 증가된 후에 적용해야 할 '
 '보험요율(이하「변경후 요 율」이라 합니다)에 대한 비율에 따라 보험금을 삭감하여 지급합니다. 다만, 증가된 위험과 관계없이 발생한 보험금'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 96},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000215',
              'chunk_char_len': 221,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.85}},
)
