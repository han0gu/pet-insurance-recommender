from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 반려동<br>물의 나이 및 품종이 정정되기 이전에는 "나이 및 품종이 정정되기 전에 적용된<br>보험요율"의 "나이 및 품종이 '
 '정정된 후에 적용해야할 보험요율"에 대한 비율에</p><br><table id=\'43\' '
 "style='font-size:14px'><thead></thead><tbody><tr><td>따라 보험금을</td><td>삭감하여 "
 "지급합니다.</td></tr></tbody></table><br><p id='44' data-category='paragraph' "
 "style='font-size:14px'>예"),
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
 'indexing': {'chunk_id': 'chunk_000868',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
