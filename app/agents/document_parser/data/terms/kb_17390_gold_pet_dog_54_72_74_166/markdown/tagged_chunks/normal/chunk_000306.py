from langchain_core.documents import Document

chunk = Document(
    page_content=('| --- | --- |\n'
 '| ∙ 절단 : 특정부위를 잘라 내는 것 ∙ 절제 : 특정부위를 잘라 없애는 것 ∙ 흡인 : 주사기 등으로 빨아들이는 것 ∙ 천자 : '
 '바늘 또는 관을 꽂아 체액․조직을 뽑아내거나 약물을 주입하는 것 | ∙ 절단 : 특정부위를 잘라 내는 것 ∙ 절제 : 특정부위를 잘라 '
 '없애는 것 ∙ 흡인 : 주사기 등으로 빨아들이는 것 ∙ 천자 : 바늘 또는 관을 꽂아 체액․조직을 뽑아내거나 약물을 주입하는 것 |'),
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
 'indexing': {'chunk_id': 'chunk_000306',
              'chunk_char_len': 237,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
