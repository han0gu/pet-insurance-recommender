from langchain_core.documents import Document

chunk = Document(
    page_content=('약 체결시에 그 타인의 서면(｢전자서명법｣제2조제2호에\n'
 '따른 전자서명이 있는 경우로서 대통령령으로 정하는 바\n'
 '에 따라 본인확인 및 위조ㆍ변조 방지에 대한 신뢰성을\n'
 '갖춘 전자문서를 포함한다)에 의한 동의를 얻어야 한다.【상법 시행령 제44조의2(타인의 생명보험)】67법 제731조제1항에 따른 본인 '
 '확인 및 위조ㆍ변조 방지에\n'
 '대한 신뢰성을 갖춘 전자문서는 다음 각 호의 요건을 모\n'
 '두 갖춘 전자문서로 한다.- 1. 전자문서에 보험금 지급사유, 보험금액, 보험계약자와\n'
 '- 보험수익자의 신원, 보험기간이 적혀 있을 것'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000083',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
