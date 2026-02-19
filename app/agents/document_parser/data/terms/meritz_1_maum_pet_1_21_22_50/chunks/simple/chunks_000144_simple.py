from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 계약자 또는 피보험자(법인인 경우에는 그 이사 또는 법인의 업무를 집행하는 그 밖의 기관)또는 이들의 법정대리인의 고의 2. 전쟁, '
 '혁명, 내란, 사변, 테러, 폭동, 소요, 노동쟁의, 기타 이들과 유사한 사태 3. 지진, 분화, 홍수, 해일 또는 이와 비슷한 천재지변 '
 '4. 핵연료물질 또는 핵연료물질에 의하여 오염된 물질의 방사성, 폭발성, 그 밖의 유해한 특성 또는 이들의 특성에 의한 사고\n'
 '【핵연료물질】사용된 연료를 포함합니다. 【핵연료물질에 의하여 오염된 물질】원자핵 분열 생성물을 포함합니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 23},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000144',
              'chunk_char_len': 283,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
