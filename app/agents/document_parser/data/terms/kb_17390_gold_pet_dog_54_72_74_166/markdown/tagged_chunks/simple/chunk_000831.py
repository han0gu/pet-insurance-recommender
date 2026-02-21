from langchain_core.documents import Document

chunk = Document(
    page_content=('- 로서 정신적 또는 육체적 훼손상태임이 의학적으로 인정되는 경우를 말한다.\n'
 '- 3) ‘치유된 후’라 함은 상해 또는 질병에 대한 치료의 효과를 기대할 수 없게\n'
 '- 되고 또한 그 증상이 고정된 상태를 말한다.\n'
 '- 4) 다만, 영구히 고정된 증상은 아니지만 치료 종결 후 한시적으로 나타나는 장\n'
 '- 해에 대하여는 그 기간이 5년 이상인 경우 해당 장해지급률의 20%를 장해지\n'
 '- 급률로 한다.\n'
 '- 5) 위 4)에 따라 장해지급률이 결정되었으나 그 이후 보장받을 수 있는 기간(계'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000831',
              'chunk_char_len': 266,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
