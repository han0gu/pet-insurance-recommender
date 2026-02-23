from langchain_core.documents import Document

chunk = Document(
    page_content=('| --- | --- | --- | --- |\n'
 '| 납입을 재촉하는 것을 말합니다. | 납입을 재촉하는 것을 말합니다. | 납입을 재촉하는 것을 말합니다. | 납입을 재촉하는 것을 '
 '말합니다. |\n'
 '제29조(보험료의 납입을 연체하여 해지된 계약의 부활(효력회복))\uf000 제28조(보험료의 납입이 연체되는 경우 납입최고(독촉)와 '
 '계약의 해지)에 따라 계약이 해지되었으나 해약환급금을 받지 않은 경우(보험계약대출 등에 따라 해약환\n'
 '급금이 차감되었으나 받지 않은 경우 또는 해약환급금이 없는 경우를 포함합니다)\n'
 '공'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000166',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
