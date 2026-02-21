from langchain_core.documents import Document

chunk = Document(
    page_content=(". 제9조(알릴 의무 위반의 효과)를 준용하여 회사가 보장을 하지 않을 수 있는<br>경우</p><br><p id='50' "
 "data-category='paragraph' style='font-size:14px'>106 KB 금쪽같은 "
 "펫보험(강아지)(무배당)(26.01)</p><br><p id='51' data-category='paragraph' "
 "style='font-size:14px'>3. 진단계약에서 보험금 지급사유가 발생할 때까지 진단을 받지 않은 경우"),
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
 'indexing': {'chunk_id': 'chunk_000876',
              'chunk_char_len': 258,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
