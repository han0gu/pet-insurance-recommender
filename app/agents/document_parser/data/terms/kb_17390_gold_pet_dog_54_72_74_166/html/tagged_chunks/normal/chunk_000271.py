from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>반한 계약을</h1><br><h1 id='86' "
 "style='font-size:14px'>말합니다.</h1><br><p id='87' data-category='paragraph' "
 "style='font-size:14px'>∙ 제척기간<br>어떤 종류의 권리에 대하여 법률이 정하고 있는 존속 기간을 말하며, 이 "
 "기간이</p><br><h1 id='88' style='font-size:14px'>지나면 권리가 소멸됩니다.</h1><br><p "
 "id='89'"),
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
 'indexing': {'chunk_id': 'chunk_000271',
              'chunk_char_len': 276,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
