from langchain_core.documents import Document

chunk = Document(
    page_content=('않았거나(보장개시 이전<br>의 원인에 의하거나 또는 그 이전에 발생한 후유장해를 포함합니다), 후유장해보<br>험금이 지급되지 않았던 '
 '피보험자에게 그 신체의 동일 부위에 또다시 제6항에 규<br>정하는 후유장해상태가 발생하였을 경우에는 직전까지의 후유장해에 대한 '
 "후유<br>장해보험금이 지급된 것으로 보고 최종 후유장해 상태에 해당되는 후유장해보험</p><br><table id='195' "
 "style='font-size:14px'><thead></thead><tbody><tr><td>금에서 이를"),
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
 'indexing': {'chunk_id': 'chunk_000344',
              'chunk_char_len': 278,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
