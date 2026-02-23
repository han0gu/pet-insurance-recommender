from langchain_core.documents import Document

chunk = Document(
    page_content=('- 4. 기타 보험수익자가 보험금의 수령 또는 보험료 납입면제 청구에 필요하여 제출하\n'
 '- 는 서류(단, 단체취급 특별약관을 부가하는 경우, 사망보험금을 지급할 때 피보험\n'
 '- 자의 법정상속인이 아닌 자가 청구하는 경우 법정상속인의 확인서 등)\n'
 '제1항 제2호의 사고증명서는 의료법 제3조(의료기관)에서 규정한 국내의 병원이나 의\n'
 '원 또는 국외의 의료관련법에서 정한 의료기관에서 발급한 것이어야 합니다.- 46 -이 법에서 의료기관이라 함은 의료인이 공중 또는 특정 '
 '다수인을 위하여 의료·조산의 업을 행하는'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
