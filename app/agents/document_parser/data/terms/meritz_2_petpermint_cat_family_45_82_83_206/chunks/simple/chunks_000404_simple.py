from langchain_core.documents import Document

chunk = Document(
    page_content=('⑥ 최초 계약의 보험계약일 이전에 이미 감염 또는 발병 한 질병 및 상해 ⑦ 원인이 어떠한 경우에도 반려동물에 대한 사료제공 또 는 급수 '
 '등 기본적인 관리에 대한 태만 ⑧ 반려동물을 범죄행위, 경주, 수색, 폭약탐지, 구조, 실험 및 이와 유사한 목적으로 이용함으로써 발생한 '
 '손해 ⑨ 수의사의 치료상의 과오로 생긴 상해 또는 질병, 수의 사 자격이 없는 자의 치료행위로 인한 비용 및 그로 인하여 가중된 비용 ⑩ '
 '국가 및 지방자치단체의 명령 또는 법률에 의한 살처 분 또는 이와 유사한 사태 ⑪ 대한민국 이외의 지역에서 발생한 사고 및'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 130},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000404',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
