from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 정당한 사유 없<br>이 이에 동의하지 않을 경우 사실 확인이 끝날 때까지 회사는 보험금 지급지연에<br>따른 이자를 지급하지 '
 "않습니다.</p><br><p id='69' data-category='list'></p><br><h1 id='70' "
 "style='font-size:16px'>용 어 풀 이 정당한 사유</h1><br><table id='71' "
 "style='font-size:16px'><thead></thead><tbody><tr><td>의무의 이행을</td><td>당사자에게"),
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
 'indexing': {'chunk_id': 'chunk_000056',
              'chunk_char_len': 277,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
