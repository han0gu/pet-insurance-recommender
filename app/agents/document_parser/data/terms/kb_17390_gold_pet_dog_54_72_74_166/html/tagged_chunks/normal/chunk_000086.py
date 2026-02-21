from langchain_core.documents import Document

chunk = Document(
    page_content=('. ∙ 연대 2인 이상이 연대하여 책임을 지므로 각자 채무의 전부를 이행할 책임을 지되 (지분만큼 분할하여 책임을 지는 것과 다름), '
 "다만 어느 1인의 이행으로 나머</td></tr></tbody></table><br><p id='108' "
 "data-category='paragraph' style='font-size:16px'>지 사람들도 책임을 면하게 되는 것을 "
 "말합니다.</p><table id='109' "
 'style=\'font-size:16px\'><thead></thead><tbody><tr><td colspan="2">제 3 관'),
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
 'indexing': {'chunk_id': 'chunk_000086',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
