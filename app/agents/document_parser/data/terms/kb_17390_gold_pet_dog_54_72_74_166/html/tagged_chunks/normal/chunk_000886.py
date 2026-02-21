from langchain_core.documents import Document

chunk = Document(
    page_content=(". 전자적 상품설명장치를 활용하여 안내한 납입최고(독촉) 등을 계약자가 모두</p><br><p id='60' "
 "data-category='paragraph' style='font-size:18px'>- 106 -</p><p id='61' "
 "data-category='list' style='font-size:16px'>수신하고 이해하였음을 확인할 것<br>3"),
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
 'indexing': {'chunk_id': 'chunk_000886',
              'chunk_char_len': 198,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
