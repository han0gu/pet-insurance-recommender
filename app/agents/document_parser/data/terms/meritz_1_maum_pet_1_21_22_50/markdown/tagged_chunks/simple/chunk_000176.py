from langchain_core.documents import Document

chunk = Document(
    page_content=('있는 단체. 다만, 사업장, 직제, 직종 등으로 구분되어 있는 경우의 단체소속 여부는\n'
 '관련법규 등에서 정하는 바에 따릅니다.# 2. 제2종 단체비영리법인단체 또는 변호사회, 의사회 등 동업자단체로서 5인 이상의 구성원이 '
 '있는\n'
 '단체# 3. 제3종 단체그 밖에 단체의 구성원을 확정시킬 수 있고 계약의 일괄적인 관리가 가능한 단체로\n'
 '서 5인 이상의 구성원이 있는 단체② 제1항의 대상단체에 소속된 자로서 동일한 보험계약을 체결한 5인 이상의 피보험자로\n'
 '피보험단체를 구성하여야 하며, 단체 구성원의 일부만을 대상으로 가입하는 경우에는'),
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
 'indexing': {'chunk_id': 'chunk_000176',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
