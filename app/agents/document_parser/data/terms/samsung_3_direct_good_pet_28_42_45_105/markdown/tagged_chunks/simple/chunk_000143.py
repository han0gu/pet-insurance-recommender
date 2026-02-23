from langchain_core.documents import Document

chunk = Document(
    page_content=('곳을 말합니다. 의료기관은 종합병원·병원·치과병원·한방병원·요양병원·정신병원·의원·치과의원·한\n'
 '의원 및 조산원으로 나누어집니다.# 제10조 (보험금 등의 지급절차)- ① 회사는 제9조(보험금 등의 청구)에서 정한 서류를 접수한 '
 '때에는 접수증을 드리고 휴\n'
 '- 대전화 문자메시지 또는 전자우편 등으로 송부하며, 그 서류를 접수한 날부터 3영업\n'
 '- 일 이내에 보험금을 지급하거나 보험료의 납입을 면제합니다.\n'
 '- ② 회사가 보험금 지급사유 또는 보험료 납입면제 사유를 조사 · 확인하기 위해 필요한'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000143',
              'chunk_char_len': 274,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
