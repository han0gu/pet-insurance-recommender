from langchain_core.documents import Document

chunk = Document(
    page_content=(". 전문금융소비자가 체결한 계약</p><p id='183' data-category='paragraph' "
 "style='font-size:14px'>62 KB 금쪽같은 펫보험(강아지)(무배당)(26.01)</p><br><h1 id='184' "
 "style='font-size:14px'>∙</h1><br><p id='185' data-category='paragraph' "
 "style='font-size:14px'>용 어 풀 이<br>전문금융소비자<br>보험계약에 관한 전문성, 자산규모 등에 비추어 보험계약에 "
 '따른 위험감수능력<br>이 있는'),
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
 'indexing': {'chunk_id': 'chunk_000151',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
