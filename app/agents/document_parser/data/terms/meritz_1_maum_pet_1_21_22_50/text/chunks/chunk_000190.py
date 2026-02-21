from langchain_core.documents import Document

chunk = Document(
    page_content=('사정이 있을 경우를 대비하여 계약을 체결할 때 또는 계약체결 이후 다음 각 호의 1에\n'
 '해당하는 자 중에서 보험금의 대리청구인(이하「지정대리청구인」이라 합니다)을 2인\n'
 '이내(2인 지정시 대표대리인을 지정)에서 지정(제4조에 따른 변경지정 포함)할 수 있습\n'
 '니다. 다만, 지정대리청구인은 보험금 청구 시에도 다음 각 호의 1에 해당하여야 합니\n'
 '다.1. 피보험자의 가족관계등록부상 또는 주민등록상의 배우자\n'
 '2. 피보험자의 3촌 이내의 친족② 제1항에도 불구하고, 지정대리청구인이 지정된 이후에 제1조(적용대상)의 보험수익자가'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
