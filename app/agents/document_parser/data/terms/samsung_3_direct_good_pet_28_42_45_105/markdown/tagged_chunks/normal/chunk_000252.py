from langchain_core.documents import Document

chunk = Document(
    page_content=('만, 해지율은 적용하지 않습니다)이 동일한 상품을 말하며, 해약환급금을 계산할\n'
 '때 기준이 되거나 비교∙안내를 위한 상품으로서 판매는 하지 않습니다.- ② 해약환급금의 지급사유가 발생한 경우 계약자는 회사에 '
 '해약환급금을 청구하여야 하\n'
 '- 며, 회사는 청구를 접수한 날부터 3영업일 이내에 해약환급금을 지급합니다. 해약환급\n'
 '- 금 지급일까지의 기간에 대한 이자의 계산은 기본계약 약관의 [별표1] 보험금을 지급\n'
 '- 할 때의 적립이율 계산을 따릅니다.'),
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
 'clause': {'clause_type': 'claim', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000252',
              'chunk_char_len': 249,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
