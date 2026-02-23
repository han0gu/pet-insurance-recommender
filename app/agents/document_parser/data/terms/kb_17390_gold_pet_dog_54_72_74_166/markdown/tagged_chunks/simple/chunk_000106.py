from langchain_core.documents import Document

chunk = Document(
    page_content=('추어 금융상품 계약에 따른 위험감수능력이 있는 금융소비자로서 다음 각\n'
 '목의 어느 하나에 해당하는 자를 말한다. 다만, 전문금융소비자 중 대통령\n'
 '령으로 정하는 자가 일반금융소비자와 같은 대우를 받겠다는 의사를 금융\n'
 '상품판매업자 또는 금융상품자문업자(이하 “금융상품판매업자등”이라\n'
 '한다)에게 서면으로 통지하는 경우 금융상품판매업자등은 정당한 사유가\n'
 '있는 경우를 제외하고는 이에 동의하여야 하며, 금융상품판매업자등이 동\n'
 '의한 경우에는 해당 금융소비자는 일반금융소비자로 본다.\n'
 '가. 국가- 나.「한국은행법」에 따른 한국은행'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000106',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
