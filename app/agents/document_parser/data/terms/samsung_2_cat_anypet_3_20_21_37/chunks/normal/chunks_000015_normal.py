from langchain_core.documents import Document

chunk = Document(
    page_content=('. 위 제4호 이외의 방사선을 쬐는 것 또는 방사능 오염 6. 최초계약의 보험개시일 이전에 이미 감염 또는 발병한 질병 및 상해 7. '
 '보험개시일로부터 그 날을 포함하여 30일 이내에 발생한 질병. 단, 이 계약이 갱신계약인 경우 에는 적용하지 않습니다. 8. 원인이 '
 '어떠한 경우에도 반려동물에 대한 사료제공 또는 급수 등 기본적인 관리에 대한 태만 9. 반려동물을 범죄행위, 경주, 수색, 폭약탐지, '
 '구조, 실험 및 이와 유사한 목적으로 이용함으로써 발생한 손해 10'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 6},
 'term_type': 'basic',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000015',
              'chunk_char_len': 261,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
