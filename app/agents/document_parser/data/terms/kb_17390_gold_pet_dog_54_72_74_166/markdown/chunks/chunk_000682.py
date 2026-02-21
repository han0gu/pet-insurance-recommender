from langchain_core.documents import Document

chunk = Document(
    page_content=('금액)을 차감한 금액을 손해액으로 간주하여 제1항에 의한 보상할 금액을 결정합\n'
 '니다.\n'
 '\uf000 피보험자가 다른 계약에 대하여 보험금 청구를 포기한 경우에도 회사의 제1항에\n'
 '의한 지급보험금 결정에는 영향을 미치지 않습니다.- \n'
 '# 용 어 풀 이# 공제계약| 유사보험으로서 공제 | 사업을 실시하는 경영주체와 | 공제 계약자 사이에 체결되 |\n'
 '| --- | --- | --- |'),
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
