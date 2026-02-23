from langchain_core.documents import Document

chunk = Document(
    page_content=('후유장해보험금에서 이미 지급받은 후유장해보험금을 차감\n'
 '하여 지급합니다. 다만,【별표2(장해분류표)】의 각 신체부\n'
 '위별 판정기준에서 별도로 정한 경우에는 그 기준에 따릅니\n'
 '다.\uf000 이미 이 보장에서 후유장해보험금 지급사유에 해당되지\n'
 '않았거나(보장개시 이전의 원인에 의하거나 또는 그 이전에\n'
 '발생한 후유장해를 포함합니다), 후유장해보험금이 지급되\n'
 '지 않았던 피보험자에게 그 신체의 동일 부위에 또다시 제6\n'
 '항에 규정하는 후유장해상태가 발생하였을 경우에는 직전까\n'
 '지의 후유장해에 대한 후유장해보험금이 지급된 것으로 보'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000016',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
