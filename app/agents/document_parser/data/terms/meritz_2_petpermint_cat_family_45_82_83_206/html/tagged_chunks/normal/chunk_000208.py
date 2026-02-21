from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 회사는 법정상속인이 보험수익자로 지정된<br>경우에는 제1항의 통지를 계약자에게 할 수 있습니다.<br>\uf000 회사는 '
 '제1항의 통지를 계약이 해지된 날부터 7일 이내<br>에 하여야 합니다.<br>\uf000 보험수익자는 통지를 받은 날(제3항에 따라 '
 '계약자에게<br>통지된 경우에는 계약자가 통지를 받은 날을 말합니다)부터<br>15일 이내에 제1항의 절차를 이행할 수 '
 "있습니다.</p><br><h1 id='76' style='font-size:18px'>【용어풀이】</h1><br><p id='77'"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000208',
              'chunk_char_len': 280,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
