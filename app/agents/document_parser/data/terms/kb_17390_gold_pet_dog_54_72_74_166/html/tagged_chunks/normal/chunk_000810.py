from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 정당한 사유<br>없이 이에 동의하지 않을 경우 사실 확인이 끝날 때까지 회사는 보험금 지급지연<br>에 따른 이자를 지급하지 '
 "않습니다.</p><br><table id='176' "
 "style='font-size:14px'><thead></thead><tbody><tr><td></td></tr><tr><td>용 어 풀 "
 '이 정당한 사유 의무의 이행을 당사자에게 기대하는 것이 무리라고 할 만한 사정이 있을 때 '
 "특</td></tr></tbody></table><br><h1 id='177'"),
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
 'indexing': {'chunk_id': 'chunk_000810',
              'chunk_char_len': 276,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
