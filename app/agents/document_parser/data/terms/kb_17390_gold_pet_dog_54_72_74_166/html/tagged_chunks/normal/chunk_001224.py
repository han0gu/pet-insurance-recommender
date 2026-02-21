from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 보험설계사 등의 행위가 없었다 하더라도 계약자 또는 피보험자가<br>사실대로 알리지 않거나 부실한 사항을 알렸다고 인정되는 '
 "경우에는 계약을<br>해지할 수 있습니다.</p><p id='20' data-category='paragraph' "
 "style='font-size:18px'>- 124 -</p><table id='21' "
 "style='font-size:16px'><thead></thead><tbody><tr><td>약자 또는 피보험자가 증명한 경우에는 "
 '보상하여 드립니다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001224',
              'chunk_char_len': 271,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
