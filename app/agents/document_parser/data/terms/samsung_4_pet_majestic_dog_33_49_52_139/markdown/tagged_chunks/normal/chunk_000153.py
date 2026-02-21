from langchain_core.documents import Document

chunk = Document(
    page_content=('- 있는 사람을 말합니다.\n'
 '- 3. 보험증권: 계약의 성립과 그 내용을 증명하기 위하여 회사가 계약자에게 드리는 증\n'
 '- 서를 말합니다.\n'
 '- 4. 진단계약: 계약을 체결하기 위하여 피보험자가 건강진단을 받아야 하는 계약을 말\n'
 '- 합니다.\n'
 '- 5. 피보험자: 보험사고의 대상이 되는 사람을 말합니다.\n'
 '# ② 지급사유 관련 용어- 1. 상해: 보험기간 중에 발생한 급격하고도 우연한 외래의 사고로 신체(의수, 의족, 의\n'
 '- 안, 의치 등 신체보조장구는 제외하나, 인공장기나 부분 의치 등 신체에 이식되어'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000153',
              'chunk_char_len': 278,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
