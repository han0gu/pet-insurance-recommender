from langchain_core.documents import Document

chunk = Document(
    page_content=('① 타인의 사망을 보험금 지급사유로 하는 계약에서 계약 을 체결할 때까지 피보험자의 서면(「전자서명법」 제 2조제2호에 따른 전자서명이 '
 '있는 경우로서 상법 시행 령에 정하는 바에 따라 본인 확인 및 위조·변조 방지 에 대한 신뢰성을 갖춘 전자문서를 포함)에 의한 동의 를 '
 '얻지 않은 경우. 다만, 단체가 규약에 따라 구성원 의 전부 또는 일부를 피보험자로 하는 계약을 체결하 는 경우에는 이를 적용하지 '
 '않습니다'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 67},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000098',
              'chunk_char_len': 229,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
