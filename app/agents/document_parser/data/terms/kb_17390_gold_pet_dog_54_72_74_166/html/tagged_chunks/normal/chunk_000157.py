from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 전문금융소비자 중 대통령<br>령으로 정하는 자가 일반금융소비자와 같은 대우를 받겠다는 의사를 금융<br>상품판매업자 또는 '
 '금융상품자문업자(이하 “금융상품판매업자등”이라<br>한다)에게 서면으로 통지하는 경우 금융상품판매업자등은 정당한 사유가<br>있는 경우를 '
 '제외하고는 이에 동의하여야 하며, 금융상품판매업자등이 동<br>의한 경우에는 해당 금융소비자는 일반금융소비자로 본다.<br>가'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000157',
              'chunk_char_len': 222,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
