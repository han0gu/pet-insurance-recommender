from langchain_core.documents import Document

chunk = Document(
    page_content=('. ∙ 최저보증이율 운용자산이익률 및 시중금리가 하락하더라도 회사에서 보증하는 최저한도의 적 용이율입니다. 예를 들어, 적립액이 '
 '공시이율에 따라 적립되며 공시이율이 0.1%인 경우(최저보증이율은 0.2%일 경우), 적립액은 공시이율(0.1%)이 아닌 '
 '최저보증이율(0.2%)로 적립됩니다. ∙ 사업방법서 회사가 보험사업의 허가를 신청할 때 첨부해야 하는 기초서류의 하나로서, 피 보험자의 '
 '범위, 보험금액 및 보험기간에 대한 제한 등이 기재된 서류를 말합니 다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
