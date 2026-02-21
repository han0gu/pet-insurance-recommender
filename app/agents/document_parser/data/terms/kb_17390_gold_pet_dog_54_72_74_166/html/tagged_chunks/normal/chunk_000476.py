from langchain_core.documents import Document

chunk = Document(
    page_content=('어 풀 이<br>∙ 절단 : 특정부위를 잘라 내는 것<br>∙ 절제 : 특정부위를 잘라 없애는 것<br>∙ 흡인 : 주사기 등으로 '
 "빨아들이는 것<br>∙ 천자 : 바늘 또는 관을 꽂아 체액․조직을 뽑아내거나 약물을 주입하는 것</p><h1 id='185' "
 "style='font-size:14px'>제6조(특별약관의 소멸)</h1><br><h1 id='186' "
 "style='font-size:14px'>피보험자가 사망하였을</h1><br><p id='187' "
 "data-category='paragraph'"),
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
 'indexing': {'chunk_id': 'chunk_000476',
              'chunk_char_len': 281,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
