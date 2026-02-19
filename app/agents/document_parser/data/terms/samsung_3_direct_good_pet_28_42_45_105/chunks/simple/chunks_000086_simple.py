from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 타인의 사망을 보험금 지급사유로 하는 계약에서 계약을 체결할 때까지 피보험자 의 서면( 「전자서명법」 제2조 제2호에 따른 '
 '전자서명이 있는 경우로서 상법 시행 령 제44조의2에 정하는 바에 따라 본인 확인 및 위조·변조 방지에 대한 신뢰성을 갖춘 전자문서를 '
 '포함)에 의한 동의를 얻지 않은 경우. 다만, 단체가 규약에 따라 구성원의 전부 또는 일부를 피보험자로 하는 계약을 체결하는 경우에는 '
 '이를 적용 하지 않습니다'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 36},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000086',
              'chunk_char_len': 235,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
