from langchain_core.documents import Document

chunk = Document(
    page_content=('- 한 특성 또는 이들의 특성에 의한 사고\n'
 '【핵연료물질】사용된 연료를 포함합니다.\n'
 '【핵연료물질에 의하여 오염된 물질】원자핵 분열 생성물을 포함합니다.- 5. 제4호 이외의 방사선을 쬐는 것 또는 방사능 오염\n'
 '- 6. 최초 계약의 보험개시일 이전에 이미 감염 또는 발병한 질병 및 상해\n'
 '- 7. 원인이 어떠한 경우에도 반려동물에 대한 사료제공 또는 급수 등 기본적인 관리에\n'
 '- 대한 태만\n'
 '- 8. 반려동물을 범죄행위, 경주, 수색, 폭약탐지, 구조, 실험 및 이와 유사한 목적으로\n'
 '- 이용함으로써 발생한 손해'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000019',
              'chunk_char_len': 283,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
