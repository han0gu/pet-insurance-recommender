from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 2인의 지정대리청<br>구인이 지정된 경우에는 그 중 대표대리인이 보험금을 청구하고 수령할 수 있으며,<br>대표대리인이 사망 '
 '등의 사유로 보험금 청구가 불가능한 경우에는 대표가 아닌 지정<br>대리청구인도 보험금을 청구하고 수령할 수 있습니다.<br>\uf000 '
 '회사가 보험금을 지정대리청구인에게 지급한 경우에는 그 이후 보험금 청구를 받더<br>라도 회사는 이를 지급하지 않습니다.</p><p '
 "id='146' data-category='paragraph' style='font-size:20px'>등</p><br><p"),
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
 'indexing': {'chunk_id': 'chunk_000305',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
