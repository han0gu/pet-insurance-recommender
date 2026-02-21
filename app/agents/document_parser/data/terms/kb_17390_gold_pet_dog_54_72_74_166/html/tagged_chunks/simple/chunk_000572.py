from langchain_core.documents import Document

chunk = Document(
    page_content=("id='57' data-category='paragraph' style='font-size:14px'>및</p><p id='58' "
 "data-category='paragraph' style='font-size:16px'>이 특별약관에서</p><br><p id='59' "
 "data-category='paragraph' style='font-size:14px'>질</p><p id='60' "
 "data-category='paragraph' style='font-size:14px'>이 특별약관에서는 보통약관 제1절 일반조항 "
 '제9조(만기환급금의'),
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
 'indexing': {'chunk_id': 'chunk_000572',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
