from langchain_core.documents import Document

chunk = Document(
    page_content=('. 이 경우 회사<br>의 요구가 있으면 계약자 및 피보험자는 이에 협력하여야 합니다.<br>④ 계약자 및 피보험자가 정당한 이유없이 '
 "제2항 및 제3항의 요구에 협조하지 않은 때에<br>는 회사는 그로 인하여 늘어난 손해는 보상하지 않습니다.</p><h1 id='76' "
 "style='font-size:14px'>제13조(합의․절충․중재․소송의 협조․대행 등)</h1><br><p id='77' "
 "data-category='paragraph' style='font-size:14px'>① 회사는 피보험자의 법률상 손해배상책임을 "
 '확정하기'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000249',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
