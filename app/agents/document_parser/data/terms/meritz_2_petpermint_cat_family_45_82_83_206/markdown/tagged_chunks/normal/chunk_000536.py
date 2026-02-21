from langchain_core.documents import Document

chunk = Document(
    page_content=('- (입을 벌림)운동, 삼킴(연하)운동 등에 따라 종합적\n'
 '- 으로 판단하여 결정한다.\n'
 '- 2) “씹어먹는 기능에 심한 장해를 남긴 때”라 함은 심\n'
 '- 한 개구(입을 벌림)운동 제한이나 저작(씹기)운동 제\n'
 '- 한으로 물이나 이에 준하는 음료 이외는 섭취하지 못\n'
 '- 하는 경우를 말한다.\n'
 '- 3) “씹어먹는 기능에 뚜렷한 장해를 남긴 때”라 함은\n'
 '- 아래의 경우 중 하나 이상에 해당되는 때를 말한다.\n'
 '- 가) 뚜렷한 개구(입을 벌림)운동 제한 또는 뚜렷한\n'
 '- 저작(씹기)운동 제한으로 미음 또는 이에 준하는'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000536',
              'chunk_char_len': 282,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
