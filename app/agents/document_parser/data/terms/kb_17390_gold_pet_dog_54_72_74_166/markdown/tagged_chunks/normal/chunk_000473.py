from langchain_core.documents import Document

chunk = Document(
    page_content=('| --- | --- |\n'
 '| 보험금 지급이 늦어지는 경우 회사가 지급할 것으로 예상되는 보험금의 일부 를 먼저 지급하는 보험금 가지급제도에 따라 먼저 지급하는 '
 '보험금을 말합니 다. | 보험금 지급이 늦어지는 경우 회사가 지급할 것으로 예상되는 보험금의 일부 를 먼저 지급하는 보험금 가지급제도에 '
 '따라 먼저 지급하는 보험금을 말합니 다. |\n'
 '- \uf000 회사는 제1항의 규정에 정한 지급기일내에 보험금을 지급하지 않았을 때(제2항의\n'
 '- 규정에서 정한 지급예정일을 통지한 경우를 포함합니다)에는 그 다음날부터 지급'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000473',
              'chunk_char_len': 282,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
