from langchain_core.documents import Document

chunk = Document(
    page_content=(". 계약자에게 전자적 상품설명장치를 활용하여 제1항에 따른 납입최고(독촉) 등</p><br><p id='59' "
 "data-category='list' style='font-size:14px'>을 한다는 사실을 미리 안내하고 동의를 받을 것<br>2"),
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
 'indexing': {'chunk_id': 'chunk_000885',
              'chunk_char_len': 137,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
